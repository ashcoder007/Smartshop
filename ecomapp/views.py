from  django.contrib import messages
from django.core.mail import send_mail
from django.shortcuts import render,redirect
from django.views.generic import TemplateView,CreateView,View,FormView,DetailView
from django.urls import reverse_lazy, reverse
from . models import *
from .forms import *
from  django.contrib.auth import authenticate,login,logout
from django.db.models import *
from django.core.paginator import Paginator
from.utils import password_reset_token
from django.conf import settings
from .recommendation_engine import get_collaborative_recommendations
from .forms import UserRatingForm
import razorpay
from django.views.decorators.csrf import csrf_exempt
class EcomMixin(object):
    def dispatch(self, request, *args, **kwargs):
        cart_id = request.session.get("cart_id")
        if cart_id:
            cart_obj = Cart.objects.get(id=cart_id)
            if request.user.is_authenticated and request.user.customer:
                cart_obj.customer = request.user.customer
                cart_obj.save()
        return super().dispatch(request, *args, **kwargs)

# Create your views here.
class Homeview(EcomMixin,TemplateView):
    template_name = "home.html"
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['product_list']=Product.objects.all().order_by("-id")
        all_products = Product.objects.all().order_by("-id")
        paginator = Paginator(all_products, 8)
        page_number = self.request.GET.get('page')
        #print(page_number)
        product_list = paginator.get_page(page_number)
        context['product_list'] = product_list
        return context
class Allproductview(EcomMixin,TemplateView):
    template_name = "allproducts.html"
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['allcategories']=Category.objects.all()
        return context

class Productdetailview(EcomMixin, TemplateView):
    template_name = "productdetail.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        url_slug = self.kwargs['slug']
        product = Product.objects.get(slug=url_slug)
        product.view_count += 1
        product.save()
        context['product'] = product

        if self.request.user.is_authenticated:
            try:
                existing_rating = UserProductRating.objects.get(user=self.request.user, product=product)
                form = UserRatingForm(instance=existing_rating)
            except UserProductRating.DoesNotExist:
                form = UserRatingForm()
            context['rating_form'] = form

        return context

    def post(self, request, *args, **kwargs):
        url_slug = self.kwargs['slug']
        product = Product.objects.get(slug=url_slug)

        if request.user.is_authenticated:
            form = UserRatingForm(request.POST)
            if form.is_valid():
                rating_obj, created = UserProductRating.objects.update_or_create(
                    user=request.user,
                    product=product,
                    defaults={'rating': form.cleaned_data['rating']}
                )
                return redirect('ecomapp:productdetail', slug=product.slug)

        return redirect('ecomapp:login')
class Addtocartview(EcomMixin,TemplateView):
    template_name = "addtocart.html"
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        product_id = self.kwargs['pro_id']
        # get product
        product_obj = Product.objects.get(id=product_id)
        # check if cart exists
        cart_id = self.request.session.get("cart_id", None)
        if cart_id:
            cart_obj = Cart.objects.get(id=cart_id)
            this_product_in_cart = cart_obj.cartproduct_set.filter(
                product=product_obj)

            # item already exists in cart
            if this_product_in_cart.exists():
                cartproduct = this_product_in_cart.last()
                cartproduct.quantity += 1
                cartproduct.subtotal += product_obj.selling_price
                cartproduct.save()
                cart_obj.total += product_obj.selling_price
                cart_obj.save()
            # new item is added in cart
            else:
                cartproduct = Cartproduct.objects.create(
                    cart=cart_obj, product=product_obj, rate=product_obj.selling_price, quantity=1, subtotal=product_obj.selling_price)
                cart_obj.total += product_obj.selling_price
                cart_obj.save()

        else:
            cart_obj = Cart.objects.create(total=0)
            self.request.session['cart_id'] = cart_obj.id
            cartproduct = Cartproduct.objects.create(
                cart=cart_obj, product=product_obj, rate=product_obj.selling_price, quantity=1, subtotal=product_obj.selling_price)
            cart_obj.total += product_obj.selling_price
            cart_obj.save()

        return context
class Managecartview(EcomMixin,TemplateView):
    def get(self, request,*args, **kwargs):
        cp_id=self.kwargs["cp_id"]
        action=request.GET.get("action")
        cp_obj = Cartproduct.objects.get(id=cp_id)
        cart_obj = cp_obj.cart

        if action == "inc":
            cp_obj.quantity += 1
            cp_obj.subtotal += cp_obj.rate
            cp_obj.save()
            cart_obj.total += cp_obj.rate
            cart_obj.save()
        elif action == "dcr":
            cp_obj.quantity -= 1
            cp_obj.subtotal -= cp_obj.rate
            cp_obj.save()
            cart_obj.total -= cp_obj.rate
            cart_obj.save()
            if cp_obj.quantity == 0:
                cp_obj.delete()

        elif action == "rmv":
            cart_obj.total -= cp_obj.subtotal
            cart_obj.save()
            cp_obj.delete()
        else:
            pass

        return redirect("ecomapp:mycart")
class Searchview(EcomMixin,TemplateView):
    template_name = "search.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        kw = self.request.GET.get("keyword")
        results = Product.objects.filter(
            Q(title__icontains=kw) | Q(description__icontains=kw) | Q(return_policy__icontains=kw))
        print(results)
        context["results"] = results
        return context
class MycartView( EcomMixin,TemplateView):
    template_name = "mycart.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        cart_id = self.request.session.get("cart_id", None)
        if cart_id:
            cart = Cart.objects.get(id=cart_id)
        else:
            cart = None
        context['cart'] = cart
        return context
class Checkoutview(EcomMixin,CreateView):
    template_name = "checkout.html"
    form_class=CheckoutForm
    success_url = reverse_lazy("ecomapp:home")
    def dispatch(self, request, *args, **kwargs):
        if request.user.is_authenticated and request.user.customer:
            pass
        else:
            return redirect("/login/?next=/checkout/")
        return super().dispatch(request, *args, **kwargs)

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        cart_id=self.request.session.get("cart_id", None)
        if cart_id:
            cart_obj = Cart.objects.get(id=cart_id)
        else:
            cart_obj=None
        context['cart']=cart_obj
        return context
    def form_valid(self, form):
        cart_id = self.request.session.get("cart_id")
        if cart_id:
            cart_obj = Cart.objects.get(id=cart_id)
            form.instance.cart = cart_obj
            form.instance.subtotal = cart_obj.total
            form.instance.discount = 0
            form.instance.total = cart_obj.total
            form.instance.order_status = "Order Received"
            del self.request.session['cart_id']
            pm = form.cleaned_data.get("payment_method")
            order = form.save()
            if pm == "Khalti":
                return redirect(reverse("ecomapp:khaltirequest") + "?o_id=" + str(order.id))
            elif pm == "Esewa":
                return redirect(reverse("ecomapp:esewarequest") + "?o_id=" + str(order.id))
        else:
            return redirect("ecomapp:home")
        return super().form_valid(form)

class Registrationview(CreateView):
    template_name = "registration.html"
    form_class = Registrationform
    success_url = reverse_lazy("ecomapp:home")

    def form_valid(self, form):
        username = form.cleaned_data.get("username")
        password = form.cleaned_data.get("password")
        email = form.cleaned_data.get("email")
        user = User.objects.create_user(username, email, password)
        form.instance.user = user
        login(self.request, user)
        return super().form_valid(form)

    def get_success_url(self):
        if "next" in self.request.GET:
            next_url = self.request.GET.get("next")
            return next_url
        else:
            return self.success_url
class Loginview(FormView):
    template_name = "login.html"
    form_class = Loginform
    success_url = reverse_lazy("ecomapp:home")

    # form_valid method is a type of post method and is available in createview formview and updateview
    def form_valid(self, form):
        uname = form.cleaned_data.get("username")
        pword = form.cleaned_data["password"]
        usr = authenticate(username=uname, password=pword)
        if usr is not None and Customer.objects.filter(user=usr).exists():
            login(self.request, usr)
        else:
            return render(self.request, self.template_name, {"form": self.form_class, "error": "cannot find account on the registered username,please register first"})

        return super().form_valid(form)

    def get_success_url(self):
        if "next" in self.request.GET:
            next_url = self.request.GET.get("next")
            return next_url
        else:
            return self.success_url

class Logoutview(View):
    def get(self, request):
        logout(request)
        return redirect("ecomapp:home")
class forgotpasswordview(FormView):
    template_name ="forgotpass.html"
    form_class= passwordforgotform
    success_url = "/forgotpassword/?m=s"

    def form_valid(self, form):
        # get email from user
        email = form.cleaned_data.get("email")
        # get current host ip/domain
        url = self.request.META['HTTP_HOST']
        # get customer and then user
        customer = Customer.objects.get(user__email=email)
        user = customer.user
        # send mail to the user with email
        text_content = 'Please Click the link below to reset your password. '
        html_content = url + "/password-reset/" + email + \
            "/" + password_reset_token.make_token(user) + "/"
        send_mail(
            'Password Reset Link | Django Ecommerce',
            text_content + html_content,
            settings.EMAIL_HOST_USER,
            [email],
            fail_silently=False,
        )
        return super().form_valid(form)
class Passwordresetview(FormView):
    template_name = "passwordreset.html"
    form_class = Passwordresetform
    success_url = "/login/"

    def dispatch(self, request, *args, **kwargs):
        email = self.kwargs.get("email")
        user = User.objects.get(email=email)
        token = self.kwargs.get("token")
        if user is not None and password_reset_token.check_token(user, token):
            pass
        else:
            return redirect(reverse("ecomapp:forgotpassword") + "?m=e")

        return super().dispatch(request, *args, **kwargs)

    def form_valid(self, form):
        password = form.cleaned_data['new_password']
        email = self.kwargs.get("email")
        user = User.objects.get(email=email)
        user.set_password(password)
        user.save()
        return super().form_valid(form)





class Aboutview(EcomMixin,TemplateView):
    template_name = "about.html"
def contact(request):
    if request.method == 'POST':
        name = request.POST.get('name')
        email = request.POST.get('email')
        message = request.POST.get('message')
        phone = request.POST.get('phone')
        contact = Contactpage(name =  name, email = email, message = message)
        contact.save()
        send_mail(
            f"Message from {name}",
            "{message}",
            "ashish232019@gmail.com",
            ['ashish232019@gmail.com'],
            fail_silently=False )
        send_mail(
            f"Hello  {name}",
            f"Thank You For Reaching Us about '{message}'\n Please Wait until Our Executive contact You \n Thank You for your Patience   -Team Ecom",
            "ashish232019@gmail.com",
            [email],
            fail_silently=False )
    return render(request, 'contact.html')
class Profileview(TemplateView):
    template_name="profile.html"
    def dispatch(self, request, *args, **kwargs):
        if request.user.is_authenticated and Customer.objects.filter(user=request.user).exists():
            pass
        else:
            return redirect("/login/?next=/profile/")
        return super().dispatch(request, *args, **kwargs)

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        customer = self.request.user.customer
        context['customer'] = customer
        orders = Order.objects.filter(cart__customer=customer).order_by("-id")
        context["orders"] = orders
        return context
class Orderdetailview(DetailView):
    template_name = "orderdetail.html"
    model = Order
    context_object_name = "ord_obj"

    def dispatch(self, request, *args, **kwargs):
        if request.user.is_authenticated and Customer.objects.filter(user=request.user).exists():
            order_id = self.kwargs["pk"]
            order = Order.objects.get(id=order_id)
            if request.user.customer != order.cart.customer:
                return redirect("ecomapp:customerprofile")
        else:
            return redirect("/login/?next=/profile/")
        return super().dispatch(request, *args, **kwargs)
class recommendation(EcomMixin,TemplateView):
    template_name = "recommendation.html"
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        if self.request.user.is_authenticated:
            user = self.request.user
            recommended_products = get_collaborative_recommendations(user)
        else:
            recommended_products = Product.objects.all().order_by('?')[:8]

        context['product_list'] = recommended_products
        return context

class RazorpayPaymentView(EcomMixin, TemplateView):
    template_name = "razorpay_payment.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        cart_id = self.request.session.get("cart_id")
        if cart_id:
            cart = Cart.objects.get(id=cart_id)
            order_amount = cart.total * 100  # Razorpay needs amount in paise
        else:
            order_amount = 50000  # fallback in case cart missing

        order_currency = 'INR'
        client = razorpay.Client(auth=(settings.RAZORPAY_API_KEY, settings.RAZORPAY_API_SECRET))

        razorpay_order = client.order.create(dict(amount=order_amount, currency=order_currency, payment_capture='1'))

        context['razorpay_order_id'] = razorpay_order['id']
        context['razorpay_merchant_key'] = settings.RAZORPAY_API_KEY
        context['amount'] = order_amount
        context['currency'] = order_currency
        return context


@csrf_exempt
def payment_success(request):
    return render(request, 'payment_success.html')





